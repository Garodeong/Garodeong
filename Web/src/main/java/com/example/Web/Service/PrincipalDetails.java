package com.example.Web.Service;

import com.example.Web.Domain.User;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.oauth2.core.user.OAuth2User;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Map;

public class PrincipalDetails implements UserDetails, OAuth2User {

   private User user;
   private Map<String, Object> attributes;

    public PrincipalDetails(User user) {
        this.user = user;
    }

   public PrincipalDetails(User user, Map<String , Object> attributes) {
       this.user = user;
       this.attributes = attributes;
   }

   @Override
   public String getName() { return null; }
   @Override
   public Map<String, Object> getAttributes() { return attributes; }

   //권한 관련 작업을 하기 위한 role return
    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
       Collection<GrantedAuthority> collections = new ArrayDeque<>();
       collections.add(() -> {
           return user.getRole().name();
       });

       return collections;
    }

    //get Password
    @Override
    public String getPassword() { return user.getPassword(); }

    //get Username
    @Override
    public String getUsername() { return user.getLoginId(); }

    //계정 만료 확인(true : 만료x)
    @Override
    public boolean isAccountNonExpired() { return true; }

    //계정 잠금확인(true : 잠금x)
    @Override
    public boolean isAccountNonLocked() { return true; }

    //비밀번호 만료 확인(true : 만료x)
    @Override
    public boolean isCredentialsNonExpired() { return true; }

    //계정 활성화 확인(true : 활성화)
    @Override
    public boolean isEnabled() {return true; }
}
