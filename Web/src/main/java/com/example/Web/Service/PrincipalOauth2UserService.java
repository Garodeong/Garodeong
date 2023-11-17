package com.example.Web.Service;

import com.example.Web.Domain.User;
import com.example.Web.Domain.UserRole;
import com.example.Web.Repository.UserRepository;
import com.example.Web.UserInfo.GoogleUserInfo;
import com.example.Web.UserInfo.KaKaoUserInfo;
import com.example.Web.UserInfo.NaverUserInfo;
import com.example.Web.UserInfo.OAuth2UserInfo;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@RequiredArgsConstructor
@Slf4j
public class PrincipalOauth2UserService extends DefaultOAuth2UserService {

    private final UserRepository userRepository;
    private final BCryptPasswordEncoder encoder;

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        OAuth2User oAuth2User = super.loadUser(userRequest);
        log.info("getAttributes : {}", oAuth2User.getAttributes());

        OAuth2UserInfo oAuth2UserInfo = null;

        String provider = userRequest.getClientRegistration().getRegistrationId();

        if(provider.equals("google")) {
            log.info("구글 로그인 요청");
            oAuth2UserInfo = new GoogleUserInfo(oAuth2User.getAttributes());
        }
        else if(provider.equals("kakao")) {
            log.info("카카오 로그인 요청");
            oAuth2UserInfo = new KaKaoUserInfo(oAuth2User.getAttributes());
        }
        else if(provider.equals("naver")) {
            log.info("네이버 로그인 요청");
            oAuth2UserInfo = new NaverUserInfo(oAuth2User.getAttributes());
        }

        String providerId = oAuth2UserInfo.getProviderId();
        String email = oAuth2UserInfo.getEmail();
        String loginId = provider + "_" + providerId;
        String nickname = oAuth2UserInfo.getName();

        Optional<User> optionalUser = userRepository.findByLoginId(loginId);
        User user = null;

        if(optionalUser.isEmpty()) {
            user = User.builder()
                    .loginId(loginId)
                    .name(nickname)
                    .name(oAuth2User.getAttribute("name"))
                    .provider(provider)
                    .providerId(providerId)
                    .role(UserRole.USER)
                    .build();
            userRepository.save(user);
        }
        else {
            user = optionalUser.get();
        }
        return new PrincipalDetails(user, oAuth2User.getAttributes());
    }
}

