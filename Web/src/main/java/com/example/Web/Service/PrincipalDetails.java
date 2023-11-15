package com.example.Web.Service;

import com.example.Web.Domain.User;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.ArrayDeque;
import java.util.Collection;

public class PrincipalDetails implements UserDetails {

   private User user;

   public PrincipalDetails(User user) { this.user = user; }

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
